(define (problem picture_54)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump blue_battery green_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear blue_battery)
        (part_at red_pump table)
        (on blue_battery green_pump)
        (part_at green_pump table)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
