(define (problem picture_210)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_pump blue_battery_1 red_pump_1 - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear blue_battery)
        (clear blue_battery_1)
        (clear red_pump_1)


        (part_at red_pump buffer_placement)
        (part_at blue_battery battery_placement)
        (part_at blue_battery_1 table)
        (part_at red_pump_1 table)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)