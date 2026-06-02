(define (problem picture_21)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump green_battery blue_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_pump)
        (clear blue_pump)
        (part_at blue_pump table)
        (part_at green_battery table)
        (part_at red_pump table)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
            (part_at green_battery battery_placement)
        )
    )
)
