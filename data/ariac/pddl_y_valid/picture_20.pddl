(define (problem picture_20)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery green_battery blue_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_battery)
        (clear blue_pump)
        (part_at blue_pump table)
        (part_at green_battery table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at red_battery battery_placement)
        )
    )
)
